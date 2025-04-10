graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1303
  arrival_time 27194.794574174357
  lifetime 184.6547452857842
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 27
    gpu 45
    rom 35
  ]
  node [
    id 1
    label "1"
    cpu 40
    gpu 42
    rom 34
  ]
  node [
    id 2
    label "2"
    cpu 35
    gpu 31
    rom 23
  ]
  node [
    id 3
    label "3"
    cpu 27
    gpu 36
    rom 36
  ]
  node [
    id 4
    label "4"
    cpu 38
    gpu 48
    rom 26
  ]
  node [
    id 5
    label "5"
    cpu 44
    gpu 47
    rom 6
  ]
  node [
    id 6
    label "6"
    cpu 43
    gpu 5
    rom 6
  ]
  node [
    id 7
    label "7"
    cpu 7
    gpu 33
    rom 32
  ]
  node [
    id 8
    label "8"
    cpu 14
    gpu 39
    rom 18
  ]
  node [
    id 9
    label "9"
    cpu 33
    gpu 18
    rom 45
  ]
  node [
    id 10
    label "10"
    cpu 40
    gpu 9
    rom 22
  ]
  node [
    id 11
    label "11"
    cpu 9
    gpu 29
    rom 9
  ]
  node [
    id 12
    label "12"
    cpu 37
    gpu 24
    rom 16
  ]
  node [
    id 13
    label "13"
    cpu 8
    gpu 30
    rom 31
  ]
  edge [
    source 0
    target 1
    bw 8
  ]
  edge [
    source 1
    target 2
    bw 10
  ]
  edge [
    source 2
    target 3
    bw 48
  ]
  edge [
    source 3
    target 4
    bw 35
  ]
  edge [
    source 4
    target 5
    bw 42
  ]
  edge [
    source 5
    target 6
    bw 30
  ]
  edge [
    source 6
    target 7
    bw 22
  ]
  edge [
    source 7
    target 8
    bw 33
  ]
  edge [
    source 8
    target 9
    bw 40
  ]
  edge [
    source 9
    target 10
    bw 44
  ]
  edge [
    source 10
    target 11
    bw 48
  ]
  edge [
    source 11
    target 12
    bw 13
  ]
  edge [
    source 12
    target 13
    bw 36
  ]
]
