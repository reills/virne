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
  id 1781
  arrival_time 39580.48491392785
  lifetime 2650.8606760477896
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 41
    gpu 30
    rom 36
  ]
  node [
    id 1
    label "1"
    cpu 40
    gpu 21
    rom 26
  ]
  node [
    id 2
    label "2"
    cpu 22
    gpu 32
    rom 42
  ]
  node [
    id 3
    label "3"
    cpu 3
    gpu 30
    rom 21
  ]
  node [
    id 4
    label "4"
    cpu 25
    gpu 20
    rom 4
  ]
  node [
    id 5
    label "5"
    cpu 21
    gpu 10
    rom 48
  ]
  node [
    id 6
    label "6"
    cpu 3
    gpu 48
    rom 6
  ]
  node [
    id 7
    label "7"
    cpu 11
    gpu 36
    rom 28
  ]
  node [
    id 8
    label "8"
    cpu 10
    gpu 26
    rom 49
  ]
  node [
    id 9
    label "9"
    cpu 5
    gpu 21
    rom 41
  ]
  node [
    id 10
    label "10"
    cpu 5
    gpu 49
    rom 4
  ]
  node [
    id 11
    label "11"
    cpu 1
    gpu 33
    rom 17
  ]
  node [
    id 12
    label "12"
    cpu 7
    gpu 27
    rom 42
  ]
  node [
    id 13
    label "13"
    cpu 18
    gpu 36
    rom 49
  ]
  edge [
    source 0
    target 1
    bw 39
  ]
  edge [
    source 1
    target 2
    bw 27
  ]
  edge [
    source 2
    target 3
    bw 31
  ]
  edge [
    source 3
    target 4
    bw 7
  ]
  edge [
    source 4
    target 5
    bw 34
  ]
  edge [
    source 5
    target 6
    bw 2
  ]
  edge [
    source 6
    target 7
    bw 45
  ]
  edge [
    source 7
    target 8
    bw 18
  ]
  edge [
    source 8
    target 9
    bw 40
  ]
  edge [
    source 9
    target 10
    bw 45
  ]
  edge [
    source 10
    target 11
    bw 18
  ]
  edge [
    source 11
    target 12
    bw 50
  ]
  edge [
    source 12
    target 13
    bw 4
  ]
]
