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
  id 1646
  arrival_time 36746.7286403226
  lifetime 2868.9527276342337
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 20
    gpu 47
    rom 29
  ]
  node [
    id 1
    label "1"
    cpu 40
    gpu 4
    rom 20
  ]
  node [
    id 2
    label "2"
    cpu 41
    gpu 0
    rom 35
  ]
  node [
    id 3
    label "3"
    cpu 46
    gpu 32
    rom 23
  ]
  node [
    id 4
    label "4"
    cpu 14
    gpu 18
    rom 20
  ]
  node [
    id 5
    label "5"
    cpu 1
    gpu 7
    rom 29
  ]
  node [
    id 6
    label "6"
    cpu 45
    gpu 11
    rom 6
  ]
  node [
    id 7
    label "7"
    cpu 46
    gpu 49
    rom 29
  ]
  node [
    id 8
    label "8"
    cpu 0
    gpu 26
    rom 23
  ]
  node [
    id 9
    label "9"
    cpu 26
    gpu 40
    rom 42
  ]
  node [
    id 10
    label "10"
    cpu 42
    gpu 17
    rom 21
  ]
  node [
    id 11
    label "11"
    cpu 43
    gpu 3
    rom 23
  ]
  node [
    id 12
    label "12"
    cpu 29
    gpu 50
    rom 50
  ]
  node [
    id 13
    label "13"
    cpu 36
    gpu 21
    rom 19
  ]
  edge [
    source 0
    target 1
    bw 21
  ]
  edge [
    source 1
    target 2
    bw 0
  ]
  edge [
    source 2
    target 3
    bw 48
  ]
  edge [
    source 3
    target 4
    bw 14
  ]
  edge [
    source 4
    target 5
    bw 41
  ]
  edge [
    source 5
    target 6
    bw 33
  ]
  edge [
    source 6
    target 7
    bw 28
  ]
  edge [
    source 7
    target 8
    bw 47
  ]
  edge [
    source 8
    target 9
    bw 8
  ]
  edge [
    source 9
    target 10
    bw 21
  ]
  edge [
    source 10
    target 11
    bw 10
  ]
  edge [
    source 11
    target 12
    bw 40
  ]
  edge [
    source 12
    target 13
    bw 50
  ]
]
